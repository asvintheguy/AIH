// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview
client<llm> Flash25 {
  provider google-ai
  options {
    model "gemini-2.5-flash-preview-04-17"
    api_key env.GEMINI_API_KEY
  }
}

client<llm> Pro25 {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.GEMINI_API_KEY
  }
}

client<llm> Grok3 {
  provider openai-generic
  options {
    model "grok-3-beta"
    api_key env.GROK_API_KEY
    base_url "https://api.x.ai/v1"
  }
}