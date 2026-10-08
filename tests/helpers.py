import asyncio

from app.events import EventEmitter


async def run_turn(container, user, session_id, message):
    """Drive one turn and return every activity event."""
    em = EventEmitter()

    async def go():
        try:
            await container.orchestrator.run_turn(user, session_id, message, em)
        finally:
            em.close()
    task = asyncio.create_task(go())
    events = [e async for e in em]
    await task
    return events


async def resume(container, user, session_id, approve):
    em = EventEmitter()

    async def go():
        try:
            await container.orchestrator.resume(user, session_id, approve, "", em)
        finally:
            em.close()
    task = asyncio.create_task(go())
    events = [e async for e in em]
    await task
    return events


def final(events):
    return next(e for e in events if e["type"] == "final")


def of(events, type_, **kw):
    return [e for e in events if e["type"] == type_ and all(e.get(k) == v for k, v in kw.items())]
