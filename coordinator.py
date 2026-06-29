import json
import os
import time
from datetime import datetime
from collections import defaultdict
import paho.mqtt.client as mqtt

BROKER_HOST = "YOUR_BROKER_IP"
BROKER_PORT = 1883

UPLINK_TOPIC = "glucose/device/+/uplink"
BROADCAST_TOPIC = "glucose/coordinator/broadcast"
LOG_FILE = os.getenv("COORDINATOR_LOG_FILE", "coordinator_log.jsonl")

STATE_COUNT = 4
RM_ALPHA = 0.2
SIGMA_BATCH_THRESHOLD = 3

mu_state = [0.0] * STATE_COUNT
Sigma_delta = [15.0] * STATE_COUNT
version = 0

devices_seen = set()
sessions_per_state = [0] * STATE_COUNT
pending_residuals = defaultdict(list)


def log_event(obj):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()


def print_status():
    print(f"[STATUS] devices={len(devices_seen)} sessions={sessions_per_state} version={version}")
    print(f"[STATUS] mu={[round(x, 4) for x in mu_state]}")
    print(f"[STATUS] sigma={[round(x, 4) for x in Sigma_delta]}")


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[MQTT] Connected with reason code: {reason_code}")
    client.subscribe(UPLINK_TOPIC)
    print(f"[MQTT] Subscribed to: {UPLINK_TOPIC}")
    print(f"[MQTT] Will broadcast to: {BROADCAST_TOPIC}")


def on_message(client, userdata, msg):
    global mu_state, Sigma_delta, version
    
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)
    except Exception as e:
        print(f"[ERROR] Failed to parse message: {e}")
        return

    device_id = data.get("device_id", "unknown")
    state = int(data.get("state", 0))
    required_offset = float(data.get("required_offset", 0))
    
    devices_seen.add(device_id)
    sessions_per_state[state] += 1

    mu_old = mu_state[state]
    mu_state[state] = mu_old + RM_ALPHA * (required_offset - mu_old)
    
    residual = required_offset - mu_state[state]
    pending_residuals[state].append(residual)
    
    if len(pending_residuals[state]) >= SIGMA_BATCH_THRESHOLD:
        values = pending_residuals[state]
        mean_sq = sum(v * v for v in values) / len(values)
        Sigma_delta[state] = max(1.0, mean_sq ** 0.5)
        pending_residuals[state].clear()
        print(f"[LEARN] State {state}: Updated sigma to {Sigma_delta[state]:.4f}")

    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device_id": device_id,
        "state": state,
        "required_offset": required_offset,
        "mu_state": list(mu_state),
        "Sigma_delta": list(Sigma_delta),
        "version": version + 1
    }
    log_event(event)

    print(f"[RX] state={state} offset={required_offset:.4f}")
    print_status()

    publish_prior(client)


def on_disconnect(client, userdata, reason_code, properties=None):
    print(f"[MQTT] Disconnected! Reason: {reason_code}")
    print("[MQTT] Will auto-reconnect...")


def publish_prior(client):
    global version
    version += 1
    
    payload = {
        "version": version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mu_state": mu_state,
        "Sigma_delta": Sigma_delta,
        "algorithm": "Robbins-Monro"
    }
    
    result = client.publish(BROADCAST_TOPIC, json.dumps(payload), qos=1)
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"[TX] Broadcasted prior v{version} to {BROADCAST_TOPIC}")
    else:
        print(f"[ERROR] Publish failed with code: {result.rc}")


def main():
    print(f"Connecting to broker: {BROKER_HOST}:{BROKER_PORT}")

    client = mqtt.Client(
        client_id="",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    connected = False
    while not connected:
        try:
            client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
            connected = True
            print("[INIT] Connected to broker successfully")
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            print("[INIT] Retrying in 3 seconds...")
            time.sleep(3)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Coordinator stopped by user")
        client.disconnect()

if __name__ == "__main__":
    main()
