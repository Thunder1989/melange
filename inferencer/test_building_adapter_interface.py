from building_adapter_interface import building_adapter_interface

target_building = 'rice'
source_buliding = ['sdh']

bl = building_adapter_interface(target_building, source_buliding)
bl.run_auto()
