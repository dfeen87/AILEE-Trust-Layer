//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { DeviceStatus, MFCDeviceTelemetry } from "../types/telemetry.js";

/**
 * EtherNet/IP (CIP) Fieldbus Parser & Serializer.
 * EtherNet/IP uses Big-Endian (Network Byte Order) for standard float32 values and uint16 register maps.
 *
 * Binary Layout (24 bytes):
 * - Byte 0-3: flowRate (Float32, Big-Endian)
 * - Byte 4-7: setpoint (Float32, Big-Endian)
 * - Byte 8-11: valvePosition (Float32, Big-Endian)
 * - Byte 12-15: temperature (Float32, Big-Endian)
 * - Byte 16-19: zeroOffset (Float32, Big-Endian)
 * - Byte 20-21: gasId (Uint16, Big-Endian)
 * - Byte 22-23: statusFlags (Uint16, Big-Endian)
 */
export class EtherNetIPAdapter {
  public static parseCIPFrame(buffer: ArrayBuffer): MFCDeviceTelemetry {
    if (buffer.byteLength < 24) {
      throw new Error(`EtherNet/IP buffer underflow: expected at least 24 bytes, got ${buffer.byteLength}`);
    }

    const view = new DataView(buffer);
    const flowRate = view.getFloat32(0, false); // false = Big-Endian
    const setpoint = view.getFloat32(4, false);
    const valvePosition = view.getFloat32(8, false);
    const temperature = view.getFloat32(12, false);
    const zeroOffset = view.getFloat32(16, false);
    const gasId = view.getUint16(20, false);
    const statusFlags = view.getUint16(22, false);

    let deviceStatus: DeviceStatus = "OK";
    if ((statusFlags & 0x8000) !== 0) {
      deviceStatus = "FAULT";
    } else if ((statusFlags & 0x4000) !== 0) {
      deviceStatus = "WARN";
    }

    return {
      flowRate,
      setpoint,
      valvePosition,
      temperature,
      zeroOffset,
      gasId,
      deviceStatus,
      statusFlags,
    };
  }

  public static serializeCIPFrame(telemetry: MFCDeviceTelemetry): ArrayBuffer {
    const buffer = new ArrayBuffer(24);
    const view = new DataView(buffer);

    view.setFloat32(0, telemetry.flowRate, false);
    view.setFloat32(4, telemetry.setpoint, false);
    view.setFloat32(8, telemetry.valvePosition, false);
    view.setFloat32(12, telemetry.temperature, false);
    view.setFloat32(16, telemetry.zeroOffset, false);
    const gasNumeric = typeof telemetry.gasId === "number" ? telemetry.gasId : parseInt(String(telemetry.gasId), 10) || 1;
    view.setUint16(20, gasNumeric, false);

    let flags = telemetry.statusFlags || 0;
    if (telemetry.deviceStatus === "FAULT") flags |= 0x8000;
    if (telemetry.deviceStatus === "WARN") flags |= 0x4000;
    view.setUint16(22, flags, false);

    return buffer;
  }
}
