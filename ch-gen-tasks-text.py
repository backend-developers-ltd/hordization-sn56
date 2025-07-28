import asyncio

from validator.core.config import load_config
from validator.cycle import process_tasks
from validator.tasks import synthetic_scheduler


async def main():
    config = load_config()
    await config.psql_db.connect()
    await synthetic_scheduler.schedule_synthetics_periodically(config)
    # await process_tasks.process_pending_tasks(config)


if __name__ == "__main__":
    asyncio.run(main())
