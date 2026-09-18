from docmind.storage.database import Database, Message


class MemoryService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def recent(self, thread_id: str, limit: int = 8) -> list[Message]:
        return self.database.messages(thread_id, limit)
