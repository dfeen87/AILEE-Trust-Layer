//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { DeviceStatus, MFCDeviceTelemetry } from "../types/telemetry.js";

/**
 * EtherCAT Compact Cyclic PDO Parser & Serializer.
 * EtherCAT (CoE / CANopen over EtherCAT) uses Little-Endian native byte order for process data objects (PDOs).
 *
 * Binary Layout (24 bytes):
 * - Byte 0-3: flowRate (Float32, Little-Endian)
 * - Byte 4-7: setpoint (Float32, Little-Endian)
 * - Byte 8-11: valvePosition (Float32, Little-Endian)
 * - Byte 12-15: temperature (Float32, Little-Endian)
 * - Byte 16-19: zeroOffset (Float32, Little-Endian)
 * - Byte 20-21: gasId (Uint16, Little-Endian)
 * - Byte 22-23: statusFlags (Uint16, Little-Endian)
 */
export class EtherCATAdapter {
  public static parsePDOFrame(buffer: ArrayBuffer): MFCDeviceTelemetry {
    if (buffer.byteLength < 24) {
      throw new Error(`EtherCAT buffer underflow: expected at least 24 bytes, got ${buffer.byteLength}`);
    }

    const view = new DataView(buffer);
    const flowRate = view.getFloat32(0, true); // true = Little-Endian
    const setpoint = view.getFloat32(4, true);
    const valvePosition = view.getFloat32(8, true);
    const temperature = view.getFloat32(12, true);
    const zeroOffset = view.getFloat32(16, true);
    const gasId = view.getUint16(20, true);
    const statusFlags = view.getUint16(22, true);

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

  public static serializePDOFrame(telemetry: MFCDeviceTelemetry): ArrayBuffer {
    const buffer = new ArrayBuffer(24);
    const view = new DataView(buffer);

    view.setFloat32(0, telemetry.flowRate, true);
    view.setFloat32(4, telemetry.setpoint, true);
    view.setFloat32(8, telemetry.valvePosition, true);
    view.setFloat32(12, telemetry.temperature, true);
    view.setFloat32(16, telemetry.zeroOffset, true);
    const gasNumeric = typeof telemetry.gasId === "number" ? telemetry.gasId : parseInt(String(telemetry.gasId), 10) || 1;
    view.setUint16(20, gasNumeric, true);

    let flags = telemetry.statusFlags || 0;
    if (telemetry.deviceStatus === "FAULT") flags |= 0x8000;
    if (telemetry.deviceStatus === "WARN") flags |= 0x4000;
    view.setUint16(22, flags, true);

    return buffer;
  }
}
