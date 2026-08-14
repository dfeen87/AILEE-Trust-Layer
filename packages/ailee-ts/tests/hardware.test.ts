//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { describe, expect, it } from "vitest";
import { MockHardwareProtocolBridge } from "../src/hardware/bridges.js";

describe("Hardware Protocol Bridges", () => {
  it("connects, sends, and receives telemetry over MockHardwareProtocolBridge", async () => {
    const bridge = new MockHardwareProtocolBridge();
    expect(bridge.isConnected()).toBe(false);

    await bridge.connect();
    expect(bridge.isConnected()).toBe(true);

    let receivedTopic = "";
    let receivedPayload: unknown = null;

    bridge.onTelemetry((topic, payload) => {
      receivedTopic = topic;
      receivedPayload = payload;
    });

    await bridge.sendTelemetry("sensors/temp", { value: 25.4 });

    expect(receivedTopic).toBe("sensors/temp");
    expect(receivedPayload).toEqual({ value: 25.4 });

    await bridge.disconnect();
    expect(bridge.isConnected()).toBe(false);
  });
});
