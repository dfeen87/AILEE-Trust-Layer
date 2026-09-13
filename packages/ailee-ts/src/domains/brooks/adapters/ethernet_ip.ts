//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import manifestJson from "../configs/sla5800_manifest.json" assert { type: "json" };
import { DeviceStatus, MFCDeviceTelemetry } from "../types/telemetry.js";

export interface FieldbusManifest {
  deviceName: string;
  frameSizeBytes: number;
  ethernetIP: {
    endianness: string;
    byteOffsets: {
      flowRate: number;
      setpoint: number;
      valvePosition: number;
      temperature: number;
      zeroOffset: number;
      gasId: number;
      statusFlags: number;
    };
  };
  etherCAT: {
    endianness: string;
    byteOffsets: {
      flowRate: number;
      setpoint: number;
      valvePosition: number;
      temperature: number;
      zeroOffset: number;
      gasId: number;
      statusFlags: number;
    };
  };
}

export const DEFAULT_MANIFEST: FieldbusManifest = manifestJson as FieldbusManifest;

/**
 * EtherNet/IP (CIP) Fieldbus Parser & Serializer.
 * EtherNet/IP uses Big-Endian (Network Byte Order) for standard float32 values and uint16 register maps.
 * Decoupled via hardware configuration manifest.
 */
export class EtherNetIPAdapter {
  public static parseCIPFrame(buffer: ArrayBuffer, manifest: FieldbusManifest = DEFAULT_MANIFEST): MFCDeviceTelemetry {
    if (buffer.byteLength < manifest.frameSizeBytes) {
      throw new Error(`EtherNet/IP buffer underflow: expected at least ${manifest.frameSizeBytes} bytes, got ${buffer.byteLength}`);
    }

    const view = new DataView(buffer);
    const offsets = manifest.ethernetIP.byteOffsets;

    const flowRate = view.getFloat32(offsets.flowRate, false); // false = Big-Endian
    const setpoint = view.getFloat32(offsets.setpoint, false);
    const valvePosition = view.getFloat32(offsets.valvePosition, false);
    const temperature = view.getFloat32(offsets.temperature, false);
    const zeroOffset = view.getFloat32(offsets.zeroOffset, false);
    const gasId = view.getUint16(offsets.gasId, false);
    const statusFlags = view.getUint16(offsets.statusFlags, false);

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

  public static serializeCIPFrame(telemetry: MFCDeviceTelemetry, manifest: FieldbusManifest = DEFAULT_MANIFEST): ArrayBuffer {
    const buffer = new ArrayBuffer(manifest.frameSizeBytes);
    const view = new DataView(buffer);
    const offsets = manifest.ethernetIP.byteOffsets;

    view.setFloat32(offsets.flowRate, telemetry.flowRate, false);
    view.setFloat32(offsets.setpoint, telemetry.setpoint, false);
    view.setFloat32(offsets.valvePosition, telemetry.valvePosition, false);
    view.setFloat32(offsets.temperature, telemetry.temperature, false);
    view.setFloat32(offsets.zeroOffset, telemetry.zeroOffset, false);
    const gasNumeric = typeof telemetry.gasId === "number" ? telemetry.gasId : parseInt(String(telemetry.gasId), 10) || 1;
    view.setUint16(offsets.gasId, gasNumeric, false);

    let flags = telemetry.statusFlags || 0;
    if (telemetry.deviceStatus === "FAULT") flags |= 0x8000;
    if (telemetry.deviceStatus === "WARN") flags |= 0x4000;
    view.setUint16(offsets.statusFlags, flags, false);

    return buffer;
  }
}
