from dataclasses import dataclass
import base64
from . import sensor_pb2
from datetime import datetime, timezone

NB_MEASUREMENTS = 6

def int_to_iso8601(value: int) -> str:
    """Convert a Unix timestamp in seconds to an ISO 8601 UTC string."""
    
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

def conversion_volt(value):
    """
    Convert raw value to voltage
    Current encoding requires sender to convert to 0-4095 range
    """
    if not value:
        return float('nan')
    return (value / 4095.0) * 3.3 # V_ref = 3.3V


@dataclass
class Sensor:
    UI: str
    time: str
    a0: float
    a1: float
    a2: float
    a3: float
    a4: float
    a5: float

def decode_proto_data(data_b64: str) -> Sensor | None:
    """
    Decodes a base64 payload containing a Protobuf SensorData message.
    Returns a Sensor object (with NaN if some values are missing).
    """
    try:
        # Decoding base64 to bytes
        data_bytes = base64.b64decode(data_b64)
        
        # Protobuf deserialization
        msg = sensor_pb2.SensorData()
        msg.ParseFromString(data_bytes)
        
        # Value extraction with NaN for missing measurements
        measurements = list(msg.measurements)
        while len(measurements) < NB_MEASUREMENTS:
            measurements.append(float('nan'))
        
        return Sensor(
            UI = msg.UI,
            time = int_to_iso8601(msg.time),
            a0=conversion_volt(measurements[0]),
            a1=conversion_volt(measurements[1]),
            a2=conversion_volt(measurements[2]),
            a3=conversion_volt(measurements[3]),
            a4=conversion_volt(measurements[4]),
            a5=conversion_volt(measurements[5])
        )
    except Exception as e:
        print("Error in Protobuf decoding:", e)
        return None