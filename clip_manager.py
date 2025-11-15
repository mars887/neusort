class ClipProcessorManager:
    """Глобальный контейнер для CLIP image/text processors."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._processor = None
            cls._instance._text_processor = None
        return cls._instance

    @property
    def processor(self):
        """Возвращает препроцессор изображений (например, CLIPProcessor или callable)."""

        return self._processor

    @processor.setter
    def processor(self, value):
        self._processor = value

    @property
    def text_processor(self):
        """Возвращает токенайзер / процессор текста для CLIP."""

        return self._text_processor

    @text_processor.setter
    def text_processor(self, value):
        self._text_processor = value

    def is_set(self):
        return self._processor is not None


# Создаем глобальный экземпляр менеджера
CLIP_PROCESSOR_MANAGER = ClipProcessorManager()
