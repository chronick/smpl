"""smpl-gen — local generation backends, a smplstream *source* tool.

Heavy ML backends (MusicGen/AudioGen/Stable-Audio) are isolated in THIS tool's own venv
(two-tier model). A dependency-free `synth` backend ships by default so generation, the
source-frame contract, and the two-tier install all work without torch.
"""

__version__ = "0.1.0"
