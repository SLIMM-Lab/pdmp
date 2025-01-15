class Sampler:

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError

    def write_data(self, folder: str, precision: int = 6):
        raise NotImplementedError