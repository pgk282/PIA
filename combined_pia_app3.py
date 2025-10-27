"""
Streamlit application that combines multiple Predictive Inventory Assistant (PIA)
prototypes into a single cohesive dashboard.  

This app exposes several tabs to help demonstrate how different pieces of the
original notebooks can come together in one place.  It includes:

* **Dashboard** – High‑level KPIs, device status cards, event stream and
  simulate buttons (derived from the original ETLSC prototype).
* **Data Capture** – Real‑time sensor readings for weight, RFID and
  vision/damage along with lists of tracked tags.
* **Forecasting & Analytics** – A Prophet forecast for the next seven days
  using recent sales history alongside summary metrics.
* **Decision & Automation** – A view into the restocking logic,
  including current stock thresholds, active alerts and logs of actions.
* **Layers** – A simple depiction of the Predictive Inventory Assistant
  architecture layers (data capture, analytics, decision & automation,
  and user interface) to help frame how each component plays a role.

This module is designed to be run with Streamlit:
    streamlit run combined_pia_app.py

Note: The original notebooks uploaded by the user are not directly accessible
in this environment, so this consolidated application recreates the key
functionality and dashboards based on the described behaviour.  Feel free to
extend the tabs or add additional charts/metrics as you see fit.
"""

import random
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import plotly.graph_objects as go

try:
    # Prophet is used for demand forecasting; fall back to a simple moving
    # average forecast if Prophet is unavailable.
    from prophet import Prophet
    from prophet.serialize import model_to_json, model_from_json
    _PROPHET_AVAILABLE = True
except Exception:
    _PROPHET_AVAILABLE = False

import streamlit as st


# -----------------------------------------------------------------------------
# Sensor simulation classes (adapted from the ETLSC prototype)
# -----------------------------------------------------------------------------

CAPACITY = 60  # capacity of on‑shelf inventory
CASE_SIZE = 12  # number of units in each backroom case
BOTTLE_WEIGHT_KG = 1.0  # weight of each unit in kilograms (simplified)


class SmartShelfSensor:
    """Simulate a weight sensor on the shelf that can infer stock and gaps."""

    def __init__(self, session: Dict[str, Any]):
        self.session = session

    def read_weight(self) -> float:
        total_units = self.session["stock"] + self.session.get("damaged", 0)
        return total_units * BOTTLE_WEIGHT_KG

    def gap_count(self) -> int:
        return max(0, CAPACITY - self.session["stock"])


class RFIDReader:
    """Simulate an RFID reader at checkout."""

    def __init__(self, session: Dict[str, Any]):
        self.session = session

    def scan_checkout(self, qty: int) -> List[str]:
        """Remove qty tags from the pool and decrement stock."""
        qty = min(qty, self.session["stock"])
        tags = []
        for _ in range(qty):
            if self.session["rfid_pool"]:
                tags.append(self.session["rfid_pool"].pop())
        self.session["stock"] -= len(tags)
        return tags


class VisionCamera:
    """Simulate a vision camera detecting gaps and damage."""

    def __init__(self, session: Dict[str, Any]):
        self.session = session

    def detect(self) -> Dict[str, int]:
        # Basic detection: damage count comes from session; gap is computed from stock
        gaps = max(0, CAPACITY - self.session["stock"])
        damaged = self.session.get("damaged", 0)
        return {"gaps": gaps, "damaged": damaged}


class IoTGateway:
    """Simulate an IoT gateway that publishes device events."""

    def __init__(self, session: Dict[str, Any]):
        self.session = session

    def publish(self, event: Dict[str, Any]) -> None:
        # Append event to the session event list
        self.session["events"].append(event)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _init_session_state() -> None:
    """Ensure all required keys exist in Streamlit session state."""
    if "stock" not in st.session_state:
        st.session_state.stock = CAPACITY  # on-shelf stock
    if "backroom_cases" not in st.session_state:
        st.session_state.backroom_cases = 8  # number of cases in backroom
    if "damaged" not in st.session_state:
        st.session_state.damaged = 0
    if "rfid_pool" not in st.session_state:
        # Generate synthetic RFID tag IDs for each unit available (shelf + backroom)
        total_units = CAPACITY + st.session_state.backroom_cases * CASE_SIZE
        st.session_state.rfid_pool = [f"TAG{i:04d}" for i in range(total_units)]
    if "events" not in st.session_state:
        st.session_state.events: List[Dict[str, Any]] = []
    if "sales_history" not in st.session_state:
        # Generate baseline sales for 45 days with simple weekday/weekend pattern
        history: List[int] = []
        for i in range(45):
            # Weekends sell slightly more
            base = 6 if (i % 7) in [5, 6] else 4
            history.append(base + random.randint(-1, 2))
        st.session_state.sales_history = history
    if "refill_in_progress" not in st.session_state:
        st.session_state.refill_in_progress = False
    if "just_refilled" not in st.session_state:
        st.session_state.just_refilled = False
    # Flag to ensure the background simulation thread only starts once
    if "simulation_thread_started" not in st.session_state:
        st.session_state.simulation_thread_started = False


def simulate_theft() -> None:
    """Simulate a theft event by decrementing stock and logging stolen tags.

    This helper removes a random number of items from the on‑shelf stock to
    represent theft. If available, RFID tags are also removed from the pool.
    A corresponding event is published to the session's event log.
    """
    # Only allow theft if there is stock on the shelf
    if st.session_state.stock > 0:
        qty = random.randint(1, 2)
        qty = min(qty, st.session_state.stock)
        stolen_tags: List[str] = []
        for _ in range(qty):
            if st.session_state.rfid_pool:
                stolen_tags.append(st.session_state.rfid_pool.pop())
        # Decrement stock
        st.session_state.stock -= qty
        # Record theft event
        event = {
            "timestamp": datetime.now().isoformat(),
            "device": "UNKNOWN",
            "event": "THEFT",
            "quantity": qty,
            "tags": stolen_tags,
        }
        gateway = IoTGateway(st.session_state)
        gateway.publish(event)


def _simulation_loop() -> None:
    """Background loop that automatically generates random inventory events.

    This loop runs continuously in a daemon thread. Every few seconds it
    randomly selects an action to simulate (purchase, damage, theft or refill).
    After updating session state, it calls st.experimental_rerun() to refresh
    the Streamlit page so that the dashboard reflects the latest state.
    """
    # Run indefinitely; the daemon thread will exit when the app terminates
    while True:
        # Sleep for a random interval between 5 and 15 seconds
        time.sleep(random.randint(5, 15))
        # Randomly choose an event to simulate
        action = random.choice(["purchase", "damage", "theft", "refill"])
        if action == "purchase":
            simulate_purchase()
        elif action == "damage":
            simulate_damage()
        elif action == "theft":
            simulate_theft()
        elif action == "refill":
            # Only refill automatically if stock is zero or below a threshold
            if st.session_state.stock <= 0:
                simulate_refill()
        # Refresh the app after each event so the UI updates
        try:
            st.experimental_rerun()
        except Exception:
            # Ignore rerun exceptions caused by re‑execution
            pass


def start_simulation_thread() -> None:
    """Start the background simulation thread if it isn't already running."""
    if not st.session_state.get("simulation_thread_started", False):
        st.session_state.simulation_thread_started = True
        # Daemon thread so it won't block Streamlit shutdown
        thread = threading.Thread(target=_simulation_loop, daemon=True)
        thread.start()


def simulate_purchase() -> None:
    """Handle purchase simulation: RFID checkout event, sales history update."""
    reader = RFIDReader(st.session_state)
    qty = random.randint(1, 2)
    tags = reader.scan_checkout(qty)
    event = {
        "timestamp": datetime.now().isoformat(),
        "device": "RFID_READER",
        "event": "CHECKOUT",
        "quantity": len(tags),
        "tags": tags,
    }
    gateway = IoTGateway(st.session_state)
    gateway.publish(event)
    # Append to sales history for forecasting
    st.session_state.sales_history.append(len(tags))


def simulate_damage() -> None:
    """Increment the damaged count and log event."""
    if st.session_state.stock > 0:
        st.session_state.damaged += 1
        event = {
            "timestamp": datetime.now().isoformat(),
            "device": "VISION_CAMERA",
            "event": "DAMAGE",
            "damaged": 1,
        }
        gateway = IoTGateway(st.session_state)
        gateway.publish(event)


def simulate_refill() -> None:
    """Refill from backroom if stock is zero or low."""
    if st.session_state.backroom_cases > 0:
        needed = CAPACITY - st.session_state.stock
        # Each case adds CASE_SIZE units, but never exceed capacity
        add_units = min(CASE_SIZE, needed)
        st.session_state.stock += add_units
        st.session_state.backroom_cases -= 1
        # Add new tags for the replenished units
        start_idx = len(st.session_state.rfid_pool)
        for i in range(add_units):
            st.session_state.rfid_pool.append(f"TAG{start_idx + i:04d}")
        event = {
            "timestamp": datetime.now().isoformat(),
            "device": "STAFF",
            "event": "REFILL",
            "quantity": add_units,
        }
        gateway = IoTGateway(st.session_state)
        gateway.publish(event)
        st.session_state.just_refilled = True


def forecast_next_7(sales: List[int]) -> pd.DataFrame:
    """Generate a 7-day forecast using Prophet or simple moving average."""
    df = pd.DataFrame({
        "ds": pd.date_range(start=datetime.now() - timedelta(days=len(sales)), periods=len(sales), freq="D"),
        "y": sales,
    })
    future_dates = pd.date_range(start=df.ds.iloc[-1] + timedelta(days=1), periods=7, freq="D")
    if _PROPHET_AVAILABLE:
        model = Prophet(daily_seasonality=False, weekly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        next_week = forecast.tail(7)[["ds", "yhat"]]
        next_week.rename(columns={"yhat": "y"}, inplace=True)
        return next_week.reset_index(drop=True)
    # Fallback: simple moving average of last 5 days
    window = 5 if len(sales) >= 5 else len(sales)
    avg = sum(sales[-window:]) / window if window > 0 else 0
    return pd.DataFrame({"ds": future_dates, "y": [avg] * 7})


def compute_kpis() -> Dict[str, Any]:
    """Compute high-level KPIs for the dashboard."""
    # Inventory accuracy and shrinkage are synthetic values for demo purposes
    accuracy = 92.0 + random.uniform(-0.5, 0.5)
    shrinkage = 1.5 + random.uniform(-0.3, 0.3)
    gap_alerts = 1 if st.session_state.stock < 15 else 0
    return {
        "Inventory Accuracy": f"{accuracy:.1f}%",
        "Shrinkage": f"{shrinkage:.1f}%",
        "Gap Alerts": gap_alerts,
        "On-Shelf Stock": st.session_state.stock,
    }


def render_dashboard() -> None:
    """Render the high-level dashboard tab."""
    kpis = compute_kpis()
    cols = st.columns(len(kpis))
    for idx, (kpi_name, value) in enumerate(kpis.items()):
        cols[idx].metric(kpi_name, value)
    st.divider()
    # Device status cards
    device_cols = st.columns(4)
    device_names = ["Smart Shelf", "RFID Reader", "Vision Camera", "Gateway"]
    for idx, name in enumerate(device_names):
        with device_cols[idx]:
            st.subheader(name)
            st.success("ONLINE")
    st.divider()
    # Action buttons (manual simulation helpers)
    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("👤 Simulate Purchase"):
            simulate_purchase()
            time.sleep(0.5)
    with action_cols[1]:
        if st.button("🧪 Simulate Damage"):
            simulate_damage()
            time.sleep(0.5)
    with action_cols[2]:
        if st.button("🔒 Simulate Theft"):
            simulate_theft()
            time.sleep(0.5)
    with action_cols[3]:
        if st.button("📦 Refill from Backroom"):
            simulate_refill()
            time.sleep(0.5)
    # Alert banners
    if st.session_state.stock == 0 and not st.session_state.refill_in_progress:
        st.session_state.refill_in_progress = True
        st.warning("Alert to Staff: Refill the shelf immediately!")
    if st.session_state.just_refilled:
        st.success("Shelf accuracy restored. Thank you!")
        st.session_state.refill_in_progress = False
        st.session_state.just_refilled = False
    # Forecast chart inline on dashboard for quick reference
    forecast_df = forecast_next_7(st.session_state.sales_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df.ds, y=forecast_df.y, mode="lines+markers", name="Forecast"))
    fig.update_layout(title="📈 7-Day Demand Forecast", xaxis_title="Date", yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)
    # Live event stream
    st.subheader("Live Event Stream")
    if st.session_state.events:
        event_df = pd.DataFrame(st.session_state.events)
        # Show last 10 events
        st.table(event_df.tail(10).iloc[::-1])
    else:
        st.info("No events yet. Simulate activity using the buttons above.")


def render_data_capture() -> None:
    """Render the Data Capture tab with sensor readings and tag lists."""
    shelf = SmartShelfSensor(st.session_state)
    vision = VisionCamera(st.session_state)
    st.subheader("Sensor Readings")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Weight (kg)", f"{shelf.read_weight():.1f}")
    with cols[1]:
        detections = vision.detect()
        st.metric("Gaps Detected", detections["gaps"])
    with cols[2]:
        st.metric("Damaged", st.session_state.damaged)
    st.divider()
    st.subheader("RFID Tag Pools")
    st.write(f"Number of tags on shelf: {st.session_state.stock}")
    st.write(f"Number of tags in backroom pool: {len(st.session_state.rfid_pool)}")
    # Show sample of tags
    if st.session_state.rfid_pool:
        st.write("Sample tags on shelf:", st.session_state.rfid_pool[-5:])
    else:
        st.info("No RFID tags available.")


def render_forecasting() -> None:
    """Render the Forecasting & Analytics tab."""
    st.subheader("Prophet Forecast (or fallback)")
    forecast_df = forecast_next_7(st.session_state.sales_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df.ds, y=forecast_df.y, mode="lines+markers", name="Forecast"))
    fig.update_layout(title="7-Day Demand Forecast", xaxis_title="Date", yaxis_title="Units Sold")
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Sales History (Last 14 Days)")
    history_df = pd.DataFrame({
        "ds": pd.date_range(start=datetime.now() - timedelta(days=len(st.session_state.sales_history) - 1), periods=len(st.session_state.sales_history), freq="D"),
        "y": st.session_state.sales_history,
    })
    # Plot last 14 days as bar chart
    fig_hist = go.Figure()
    recent = history_df.tail(14)
    fig_hist.add_trace(go.Bar(x=recent.ds, y=recent.y, name="Units Sold"))
    fig_hist.update_layout(xaxis_title="Date", yaxis_title="Units Sold", bargap=0.2)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.divider()
    st.info("This tab generates a 7-day demand forecast using Prophet if available. Otherwise, a simple moving average forecast is used.")


def render_decision_automation() -> None:
    """Render the Decision & Automation tab."""
    st.subheader("Decision Logic")
    gap_threshold = st.slider("Low Stock Threshold", min_value=1, max_value=20, value=15)
    # Show current status and whether an alert is active
    if st.session_state.stock <= 0:
        st.error("Stock is depleted! Immediate refill required.")
    elif st.session_state.stock <= gap_threshold:
        st.warning(f"Stock below threshold ({gap_threshold}). Please schedule a refill soon.")
    else:
        st.success("Stock level within acceptable range.")
    st.divider()
    st.subheader("Restocking History")
    # Filter and display refill events
    refills = [e for e in st.session_state.events if e["event"] == "REFILL"]
    if refills:
        refill_df = pd.DataFrame(refills)
        st.table(refill_df.tail(10).iloc[::-1])
    else:
        st.info("No refill events recorded yet.")


def render_layers() -> None:
    """Render a simple depiction of the PIA architecture layers."""
    st.subheader("PIA Architecture Layers")
    descriptions = [
        ("Data Capture", "Collects data from the Smart Shelf, RFID Reader, and Vision Camera."),
        ("Analytics", "Processes sales history and predicts future demand using models like Prophet."),
        ("Decision & Automation", "Determines when to reorder and triggers refill actions based on thresholds."),
        ("User Interface", "Presents dashboards, KPIs, and controls for staff interaction."),
    ]
    for layer, desc in descriptions:
        st.markdown(f"### {layer}")
        st.write(desc)
        st.divider()


def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(page_title="Combined PIA Demo", page_icon="🛒", layout="wide")
    _init_session_state()
    # Automatically start the background simulation thread once session state is initialized
    start_simulation_thread()
    st.title("Combined Predictive Inventory Assistant Dashboard")
    st.markdown(
        "This app consolidates features from multiple prototypes into a unified interface."
    )
    # Create tabs for different views
    tabs = st.tabs([
        "Dashboard",
        "Data Capture",
        "Forecasting & Analytics",
        "Decision & Automation",
        "Layers",
    ])
    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        render_data_capture()
    with tabs[2]:
        render_forecasting()
    with tabs[3]:
        render_decision_automation()
    with tabs[4]:
        render_layers()


if __name__ == "__main__":
    main()