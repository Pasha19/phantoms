import numpy as np

from FixedResampleVolume import FixedResampleVolume


array = np.array(
    [
        [
            [1, 1],
            [1, 1],
        ],
        [
            [2, 2],
            [2, 2],
        ],
    ],
    dtype=np.float32,
)

volume = FixedResampleVolume(array, spacing=(10, 10, 10))
print(volume.bounds())

resampled_volume = volume.resample(new_spacing=[4, 4, 4], interpolation=1)
print(resampled_volume.bounds())

resampled_array = resampled_volume.tonumpy()

print(resampled_array)
