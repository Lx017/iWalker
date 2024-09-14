import numpy as np
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
import numpy as np
import json


class SharedNp:
    def __init__(self, config_path):
        self.dict = json.load(open(config_path, "r"))
        self.arrays = {}
        self.total_size = 0

        # Calculate the total size needed for the shared memory block
        for sm_name, info in self.dict.items():
            shape, type = info.split(";")
            shape = [int(i) for i in shape.split("x")]
            self.dict[sm_name] = (shape, type)
            self.total_size += np.prod(shape) * np.dtype(type).itemsize

        # Create shared memory block
        try:
            self.shm = shared_memory.SharedMemory(
                create=True, size=self.total_size, name="myshm_" + config_path
            )
            if_create = True
            print(f"Shared memory block created with size {self.total_size}")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name="myshm_" + config_path)
            if_create = False
            if self.shm.size != self.total_size:
                self.shm.close()
                self.shm.unlink()  # delete the shared memory
                raise ValueError(
                    "Shared memory block already exists with a different size, please restart the program"
                )
            print(
                f"Shared memory block already exists with size {self.total_size}, attaching to it"
            )

        # Create numpy arrays linked to the shared memory
        offset = 0
        for sm_name, (shape, dtype) in self.dict.items():
            dtype = np.dtype(dtype)
            size = np.prod(shape) * dtype.itemsize
            arr = np.ndarray(
                shape, dtype=dtype, buffer=self.shm.buf[offset : offset + size]
            )
            self.arrays[sm_name] = arr
            if if_create:
                arr.fill(0)  # Initialize the array with zeros
            offset += size

    def __getitem__(self, sm_name):
        if sm_name in self.arrays:
            return self.arrays[sm_name]
        else:
            raise KeyError(f"No shared memory array found with name '{sm_name}'")

    def list(self):
        return self.dict.keys()


if __name__ == "__main__":
    setter = SharedNp("shm1.json")

    # Access and modify the numpy arrays
    setter["sm1"][0, 0] = 1.5
    setter["sm2"][1, 1] = 42

    getter = SharedNp("shm1.json")

    print(getter["sm1"])
    print(getter["sm2"])
