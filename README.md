# LangChain Coral agent template
This repo contains an example agent that runs on Coral. It's built with LangChain, and it runs in a loop with memory of what it does.

Coral agents don't need to be anything special, though they perform best when they can remember their own actions.

When building systems with Coral, their lifecycles are scoped to a session, so the agents can retain memory of their actions
 without needing to worry so much about running out of context window space.

This example also shows how to integrate with Coral's payment system and get paid for your agents.

## Features
* Short-term memory
* Claiming for results and work done to earn money
* Working with Coral messages via [MCP Resources](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)
which allows agents to instantly receive new messages addressed to them.
* Easy integration points for adding your own tools and MCP servers to the agent


## Developing with this
### Requirements
1. Python 3.13+
2. OpenAI API key or compatible API key (e.g. openrouter) 
3. Coral server running locally

### Running in devmode
[Devmode](https://docs.coralprotocol.org/guides/writing-agents#devmode) allows you to run an agent using your preferred tooling and it will connect to a yet to exist session, where the server will see devmode is enabled and create the session for you.
This is the recommended method of developing indvidual agents since it saves you from needing to manually create sessions to test out each iteration.
See the [docs](https://docs.coralprotocol.org/guides/writing-agents#devmode) for more info.

This template will load the local .env when in devmode, determined by CORAL_ORCHESTRATION_RUNTIME not being set (the coral server would set it outside of devmode)


## Support & Resources
Docs: [Coral Protocol Documentation](https://docs.coralprotocol.org/)
Community: [Coral Discord](https://discord.gg/MqcwYy6gxV)

