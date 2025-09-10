from .bc import BCAgent
from .calql import CalQLAgent
from .cql import CQLAgent
from .iql import IQLAgent
from .sac import SACAgent
from .closed_loop_sac import ClosedLoopSACAgent
from .sac_bc import SACBCWithTargetAgent

agents = {
    "bc": BCAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
    "calql": CalQLAgent,
    "sac": SACAgent,
    "closed_loop_sac": ClosedLoopSACAgent,
    "sac_bc": SACBCWithTargetAgent,
}
