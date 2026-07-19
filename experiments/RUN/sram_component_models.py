"""Generic digital component adapters used by the SRAM-CIM experiment configs."""

from hwcomponents import ActionCost, ComponentModel, action
from hwcomponents_library.library.aladdin import AladdinAdder, AladdinRegister


class DigitalSramShiftAccumulator(ComponentModel):
    """Register-plus-adder proxy for one bit-serial accumulation step."""

    component_name = ["DigitalSramShiftAccumulator"]
    priority = 0.9

    def __init__(self, tech_node: float, width: int = 25):
        self.adder = AladdinAdder(tech_node=tech_node, width=width)
        self.register = AladdinRegister(tech_node=tech_node, width=width)
        super().__init__(subcomponents=[self.adder, self.register])

    @action
    def read(self) -> ActionCost:
        add = self.adder.add()
        write = self.register.write()
        return ActionCost(
            energy=add.energy + write.energy,
            throughput=min(add.throughput, write.throughput),
            latency=add.latency + write.latency,
        )
