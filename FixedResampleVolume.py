from typing_extensions import Self

import vedo
from vedo import utils
import vedo.vtkclasses as vtki


class FixedResampleVolume(vedo.Volume):
    def resample(self, new_spacing: list[float], interpolation=1) -> Self:
        """
        Resamples a `Volume` to be larger or smaller.

        This method modifies the spacing of the input.
        Linear interpolation is used to resample the data.

        Arguments:
            new_spacing : (list)
                a list of 3 new spacings for the 3 axes
            interpolation : (int)
                0=nearest_neighbor, 1=linear, 2=cubic
        """
        rsp = vtki.new("ImageResample")
        oldsp = self.spacing()
        for i in range(3):
            if oldsp[i] != new_spacing[i]:
                rsp.SetAxisOutputSpacing(i, new_spacing[i])
        rsp.InterpolateOn()
        rsp.SetInterpolationMode(interpolation)
        rsp.OptimizationOn()
        rsp.SetInputData(self.dataset)
        rsp.Update()
        self._update(rsp.GetOutput())
        self.pipeline = utils.OperationNode(
            "resample", comment=f"spacing: {tuple(new_spacing)}", parents=[self], c="#4cc9f0"
        )
        return self
