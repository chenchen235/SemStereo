from .SemStereo import SemStereo
from .SemStereo_WHU import SemStereo_WHU
from .loss import model_label_loss, model_loss_train, model_loss_test, LRSC_loss


__models__ = {
    "SemStereo": SemStereo,
    "SemStereo_WHU": SemStereo_WHU
}
