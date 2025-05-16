import numpy as np

raw = open("blurred_1024y_1024x_4c.bytes", "rb").read()
print(f"Read {raw.__len__()} bytes")

linear = np.frombuffer(raw, dtype=np.float32)

reshaped = linear.reshape((1024, 1024, 4))
print(reshaped.shape)
