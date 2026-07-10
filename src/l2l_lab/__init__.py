import logging

from l2l_lab._utils.logging_utils import GutterFormatter, _DynamicStdoutHandler

logger = logging.getLogger("l2l_lab")
_handler = _DynamicStdoutHandler()
_handler.setFormatter(GutterFormatter())
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
