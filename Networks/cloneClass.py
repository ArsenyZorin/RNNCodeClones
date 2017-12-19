class CloneClass:

    def __init__(self, base_class):
        self.base_class = base_class
        self.clones = []

    def append(self, clone):
        self.clones.append(clone)
