import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s\n', datefmt='%H:%M:%S')

from typing import Union, BinaryIO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines import run_tasks, Task
from cognee.tasks.ingestion import ingest_data, resolve_data_directories
from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)
from uuid import uuid5, NAMESPACE_OID

# Set up logger at the top of the file
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def add(
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_name: str = "main_dataset",
    user: User = None,
):
    logger.debug("Starting add function with parameters:")
    logger.debug(f"dataset_name: {dataset_name}")
    logger.debug(f"user: {user}")
    logger.debug(f"data type: {type(data)}")

    try:
        # Create tables for databases
        logger.debug("Creating relational database tables")
        await create_relational_db_and_tables()
        logger.debug("Creating pgvector database tables")
        await create_pgvector_db_and_tables()

        # Initialize first_run attribute if it doesn't exist
        if not hasattr(add, "first_run"):
            logger.debug("Initializing first_run attribute")
            add.first_run = True

        if add.first_run:
            logger.debug("First run detected, testing connections")
            from cognee.infrastructure.llm.utils import test_llm_connection, test_embedding_connection

            logger.debug("Testing LLM connection")
            await test_llm_connection()
            logger.debug("Testing embedding connection")
            await test_embedding_connection()
            add.first_run = False
            logger.debug("First run completed")

        if user is None:
            logger.debug("No user provided, getting default user")
            user = await get_default_user()
            logger.debug(f"Default user: {user}")

        logger.debug("Setting up pipeline tasks")
        tasks = [Task(resolve_data_directories), Task(ingest_data, dataset_name, user)]

        dataset_id = uuid5(NAMESPACE_OID, dataset_name)
        logger.debug(f"Generated dataset_id: {dataset_id}")

        logger.debug("Starting pipeline execution")
        pipeline = run_tasks(
            tasks=tasks, dataset_id=dataset_id, data=data, pipeline_name="add_pipeline"
        )

        async for result in pipeline:
            logger.debug(f"Pipeline result: {result}")
            print(result)

    except Exception as e:
        logger.exception("Error in add function")
        raise
