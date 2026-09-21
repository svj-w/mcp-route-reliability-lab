# Three-Tool Agent Learning Notes

This checkpoint separates the agent protocol from the reliability benchmark. The
throwaway exercise used three local tools—`calculator`, `weather`, and `time`—and
a hand-authored model response, so it required no API key and made no network call.

## Simulated success

User query:

```text
What is the weather in Shenzhen?
```

Tool schemas exposed to the model:

```json
[
  {"name": "calculator", "parameters": {"expression": "string"}},
  {"name": "weather", "parameters": {"city": "string"}},
  {"name": "time", "parameters": {"timezone": "string"}}
]
```

Simulated model decision:

```json
{"name": "weather", "arguments": {"city": "Shenzhen"}}
```

The host validated the JSON, called a deterministic weather stub, and returned
`{"city":"Shenzhen","condition":"sunny","temperature_c":28}`. Only then could a
model turn that tool result into a user-facing answer.

## Simulated routing failure

For `What time is it in Tokyo?`, a deliberately incorrect response selected
`weather(city="Tokyo")`. The tool call was syntactically valid but semantically
wrong. Argument validation cannot repair a wrong tool choice; routing reliability
must be evaluated before execution.

## Boundaries learned

- `Query → Tool name + arguments` is a structured prediction problem.
- Tool selection and tool execution are separate failure surfaces.
- JSON Schema validates shape, not whether the selected capability matches intent.
- HTTP transports requests and results but does not provide routing confidence.
- A reliability layer can abstain before execution, while the executor still owns
  authentication, side effects, retries, and tool-result handling.

