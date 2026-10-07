import asyncio
from main import invoke

class Context:
    def __init__(self, session_id):
        self.session_id = session_id

async def run_test():
    # Session 1: Alice reveals her favorite cloud
    ctx_alice = Context("session-alice")
    async for _ in invoke({"prompt": "My favorite cloud provider is AWS."}, ctx_alice):
        pass

    # Session 2: Bob asks what Alice's cloud is
    ctx_bob = Context("session-bob")
    bob_resp = ""
    async for event in invoke({"prompt": "What is my favorite cloud provider?"}, ctx_bob):
        if "contentBlockDelta" in event.get("event", {}):
            bob_resp += event["event"]["contentBlockDelta"].get("delta", {}).get("text", "")
    print("Bob Response (Must be unaware):", bob_resp)

    # Session 1: Alice asks what she said earlier
    alice_resp = ""
    async for event in invoke({"prompt": "What cloud provider did I say I like?"}, ctx_alice):
        if "contentBlockDelta" in event.get("event", {}):
            alice_resp += event["event"]["contentBlockDelta"].get("delta", {}).get("text", "")
    print("Alice Response (Must recall AWS):", alice_resp)

asyncio.run(run_test())