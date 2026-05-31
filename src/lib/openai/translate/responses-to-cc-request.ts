import type { ChatCompletionsPayload, ChatCompletionResponse } from "~/types/api/openai-chat-completions"
import type { ResponsesPayload, ResponsesResponse } from "~/types/api/openai-responses"

/**
 * Translates an incoming Responses API request payload (/v1/responses)
 * into an OpenAI-compatible Chat Completions payload structure.
 */
export function translateResponsesToChatCompletions(payload: ResponsesPayload): ChatCompletionsPayload {
  const messages: Array<unknown> = []

  // 1. Process System / Developer Instructions
  if (payload.instructions) {
    messages.push({
      role: "system",
      content: payload.instructions,
    })
  } else if ((payload as Record<string, unknown>).developer_instructions) {
    messages.push({
      role: "system",
      content: (payload as Record<string, unknown>).developer_instructions,
    })
  }

  // 2. Process Input Payload Structure
  if (typeof payload.input === "string") {
    messages.push({
      role: "user",
      content: payload.input,
    })
  } else if (Array.isArray(payload.input)) {
    for (const item of payload.input) {
      if (typeof item === "string") {
        messages.push({ role: "user", content: item })
      } else {
        const itemObj = item as Record<string, unknown>
        if (itemObj.role && itemObj.content) {
          messages.push({
            role: itemObj.role === "developer" ? "system" : itemObj.role,
            content: translateContentParts(itemObj.content),
          })
        } else if (itemObj.type === "text" || itemObj.input_text) {
          messages.push({
            role: "user",
            content: itemObj.text || itemObj.input_text,
          })
        }
      }
    }
  }

  const rawPayload = payload as Record<string, unknown>
  const effort = rawPayload.model_reasoning_effort || rawPayload.reasoning_effort

  // 3. Construct Standard Chat Completions Payload Matrix
  return {
    model: payload.model,
    messages: messages as ChatCompletionsPayload["messages"],
    tools: payload.tools ? translateToolsToCC(payload.tools) : undefined,
    stream: Boolean(payload.stream),
    temperature: payload.temperature,
    max_tokens: payload.max_tokens,
    reasoning_effort: effort as ChatCompletionsPayload["reasoning_effort"],
  }
}

/**
 * Translates a complete non-streaming Chat Completions response object back
 * into the standard format expected by a Responses API client wrapper.
 */
export function translateCCToResponsesResponse(ccResponse: ChatCompletionResponse): ResponsesResponse {
  const choice = ccResponse.choices[0]
  const message = choice.message
  const contentText = message.content || ""

  const outputItem = {
    id: `item_${Math.random().toString(36).slice(2, 11)}`,
    type: "message",
    role: "assistant",
    content: [
      {
        type: "text",
        text: contentText,
      },
    ],
  }

  // Convert functional tool calls back if applicable
  if (message.tool_calls && message.tool_calls.length > 0) {
    for (const tc of message.tool_calls) {
      outputItem.content.push({
        type: "tool_call",
        id: tc.id,
        name: tc.function.name,
        arguments: tc.function.arguments,
      } as unknown as { type: "text"; text: string })
    }
  }

  return {
    id: ccResponse.id.replace("chatcmpl-", "resp_"),
    object: "response",
    status: "completed",
    model: ccResponse.model,
    output_text: contentText,
    output_item: outputItem as ResponsesResponse["output_item"],
    usage: ccResponse.usage as ResponsesResponse["usage"],
  }
}

/**
 * An Async Generator that captures individual incoming Chat Completion stream chunks
 * and normalizes them on-the-fly into structured Responses API SSE events.
 */
export async function* translateCCStreamToResponsesStream(
  ccStream: AsyncIterable<unknown>,
): AsyncGenerator<{ event: string; data: string }, void, unknown> {
  const responseId = `resp_${Math.random().toString(36).slice(2, 11)}`
  const itemId = `item_${Math.random().toString(36).slice(2, 11)}`

  // Emit Initial Lifecycle Stream Sequences
  yield {
    event: "response.created",
    data: JSON.stringify({ id: responseId, object: "response", status: "in_progress" }),
  }

  yield {
    event: "response.output_item.added",
    data: JSON.stringify({
      response_id: responseId,
      output_index: 0,
      item: { id: itemId, type: "message", role: "assistant", content: [] },
    }),
  }

  const partIndex = 0

  for await (const chunk of ccStream) {
    let rawChunk = chunk
    if (typeof chunk === "string") {
      try {
        if (chunk.trim() === "[DONE]") continue
        rawChunk = JSON.parse(chunk)
      } catch {
        continue
      }
    }

    const chunkObj = rawChunk as Record<string, unknown>
    const choices = chunkObj.choices as Array<Record<string, unknown>> | undefined
    const choice = choices?.[0]
    const delta = choice?.delta as Record<string, unknown> | undefined

    // Handle Text Chunk Generation Deltas
    if (delta?.content) {
      yield {
        event: "response.content_part.delta",
        data: JSON.stringify({
          response_id: responseId,
          output_index: 0,
          part_index: partIndex,
          delta: {
            type: "text_delta",
            text: delta.content,
          },
        }),
      }
    }

    // Handle Tool Invocation Stream Deltas
    if (delta?.tool_calls) {
      const toolCalls = delta.tool_calls as Array<Record<string, unknown>>
      for (const tc of toolCalls) {
        const fn = tc.function as Record<string, unknown> | undefined
        yield {
          event: "response.content_part.delta",
          data: JSON.stringify({
            response_id: responseId,
            output_index: 0,
            part_index: partIndex,
            delta: {
              type: "tool_call_delta",
              id: tc.id,
              name: fn?.name,
              arguments: fn?.arguments,
            },
          }),
        }
      }
    }
  }

  // Close Down Stream State Lifecycle Events
  yield {
    event: "response.output_item.done",
    data: JSON.stringify({
      response_id: responseId,
      output_index: 0,
      item: { id: itemId, status: "completed" },
    }),
  }

  yield {
    event: "response.done",
    data: JSON.stringify({ id: responseId, status: "completed" }),
  }
}

/* Helper Parsing Subroutines */

function translateContentParts(content: unknown): unknown {
  if (typeof content === "string") return content
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (part && typeof part === "object") {
        const partObj = part as Record<string, unknown>
        if (partObj.type === "input_text" || partObj.type === "text") {
          return { type: "text", text: partObj.text || partObj.input_text }
        }
        if (partObj.type === "input_image" || partObj.type === "image_url") {
          return {
            type: "image_url",
            image_url: partObj.image_url || { url: partObj.input_image },
          }
        }
      }
      return part
    })
  }
  return content
}

function translateToolsToCC(tools: Array<unknown>): ChatCompletionsPayload["tools"] {
  return tools.map((tool) => {
    if (tool && typeof tool === "object") {
      const toolObj = tool as Record<string, unknown>
      if (toolObj.type === "function") return tool as ChatCompletionsPayload["tools"] extends Array<infer T> ? T : never
    }
    return tool as ChatCompletionsPayload["tools"] extends Array<infer T> ? T : never
  })
}
