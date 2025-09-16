class ClipProcessorManager:
    """
    Singleton-класс для управления глобальным CLIP_PROCESSOR.
    Гарантирует, что все части программы получают одну и ту же актуальную ссылку.
    """
    _instance = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClipProcessorManager, cls).__new__(cls)
        return cls._instance

    @property
    def processor(self):
        return self._processor

    @processor.setter
    def processor(self, value):
        self._processor = value

    def is_set(self):
        return self._processor is not None

# Создаем глобальный экземпляр менеджера
CLIP_PROCESSOR_MANAGER = ClipProcessorManager()