import infrastructure.config as config
from database import DatabaseConnection

class MemoryStore:
    def __init__(self):
        self.database = DatabaseConnection()
