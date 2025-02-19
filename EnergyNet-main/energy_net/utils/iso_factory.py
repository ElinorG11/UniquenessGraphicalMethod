# iso_factory.py

from typing import Dict, Any
from energy_net.dynamics.iso.hourly_pricing_iso import HourlyPricingISO
from energy_net.dynamics.iso.dynamic_pricing_iso import DynamicPricingISO
from energy_net.dynamics.iso.quadratic_pricing_iso import QuadraticPricingISO
from energy_net.dynamics.iso.random_pricing_iso import RandomPricingISO
from energy_net.dynamics.iso.time_of_use_pricing_iso import TimeOfUsePricingISO
from energy_net.dynamics.iso.fixed_pricing_iso import FixedPricingISO 
from energy_net.dynamics.iso.iso_base import ISOBase


def iso_factory(iso_type: str, iso_parameters: Dict[str, Any]) -> ISOBase:
    """
    Factory function to create ISO instances based on the iso_type.
    
    Args:
        iso_type (str): The type of ISO to create.
        iso_parameters (Dict[str, Any]): Parameters required to instantiate the ISO.
    
    Returns:
        ISOBase: An instance of the specified ISO.
    
    Raises:
        ValueError: If the iso_type is unknown.
    """
    iso_type = iso_type.strip()
    iso_type_mapping = {
        'HourlyPricingISO': HourlyPricingISO,
        'DynamicPricingISO': DynamicPricingISO,
        'QuadraticPricingISO': QuadraticPricingISO,
        'RandomPricingISO': RandomPricingISO,
        'TimeOfUsePricingISO': TimeOfUsePricingISO,
        'FixedPricingISO': FixedPricingISO 
    }
    
    if iso_type in iso_type_mapping:
        iso_class = iso_type_mapping[iso_type]
        try:
            if iso_type == 'FixedPricingISO':
                pricing_schedule = iso_parameters.get('pricing_schedule', None)
                if pricing_schedule is None:
                    raise ValueError("FixedPricingISO requires a 'pricing_schedule' parameter.")
                return iso_class(pricing_schedule=pricing_schedule)
            else:
                return iso_class(**iso_parameters)
        except TypeError as e:
            raise TypeError(f"Error initializing {iso_type}: {e}") from e
    else:
        raise ValueError(f"Unknown ISO type: {iso_type}")
