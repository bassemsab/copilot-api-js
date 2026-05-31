import { describe, expect, test } from "bun:test"

import type { ChatCompletionResponse } from "~/types/api/openai-chat-completions"
import type { ResponsesPayload } from "~/types/api/openai-responses"

import {
  translateCCToResponsesResponse,
  translateResponsesToChatCompletions,
} from "~/lib/openai/translate/responses-to-cc-request"

describe("Inbound Responses to Chat Completions Request Translation", () => {
  test("translateResponsesToChatCompletions maps instructions and simple text input items", () => {
    const mockIncomingPayload = {
      model: "custom-local-model",
      instructions: "Act as a specialized coding assistant.",
      input: "Write a quicksort in TypeScript",
      stream: false,
      temperature: 0.7,
    }

    const result = translateResponsesToChatCompletions(mockIncomingPayload as unknown as ResponsesPayload)

    expect(result.model).toBe("custom-local-model")
    expect(result.stream).toBe(false)
    expect(result.temperature).toBe(0.7)
    expect(result.messages).toEqual([
      { role: "system", content: "Act as a specialized coding assistant." },
      { role: "user", content: "Write a quicksort in TypeScript" },
    ])
  })

  test("translateCCToResponsesResponse maps static choice structures back to responses wrapper format", () => {
    const mockChatCompletionResponse = {
      id: "chatcmpl-test789",
      model: "custom-local-model",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "Here is your response payload.",
          },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 40, completion_tokens: 20, total_tokens: 60 },
    }

    const result = translateCCToResponsesResponse(mockChatCompletionResponse as unknown as ChatCompletionResponse)

    expect(result.id).toBe("resp_test789")
    expect(result.status).toBe("completed")
    expect(result.output_text).toBe("Here is your response payload.")
    expect(result.output_item.content[0]).toEqual({
      type: "text",
      text: "Here is your response payload.",
    })
  })
})
