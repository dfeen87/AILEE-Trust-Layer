# Windows operating manual (v9.4.0)

## Windows-native boundary

`WindowsPlatformAdapter` activates only on Win32 and does not use a POSIX
emulation layer. `identity()` uses `OpenProcessToken(TOKEN_QUERY)` to report
whether the current process token is queryable, closes the token handle, and
reports PID and the environment username. This is execution context evidence,
not principal authentication, impersonation, or privilege elevation.

| Capability group | Classification on Windows | Mechanism |
|---|---|---|
| file read/write/delete, directory traversal, protected-resource read | `SUPPORTED_WITH_LIMITATIONS` | Windows-backed `pathlib` file APIs; ACL/sharing rules apply |
| tool/command/subprocess creation | `SUPPORTED_WITH_LIMITATIONS` | direct argv through `subprocess.run`; current token inherited |
| process control | `SUPPORTED_WITH_LIMITATIONS` | `OpenProcess(PROCESS_TERMINATE)` then `TerminateProcess`; handle closed |
| network connect/destination access | `SUPPORTED_WITH_LIMITATIONS` | outbound Winsock-backed TCP connect |
| privilege-sensitive, resource allocation, hardware request | `OBSERVABLE_ONLY` | context is not granted or confined |
| credential-sensitive operation | `UNAVAILABLE` | no enforcement implementation |

No capability is unconditionally `SUPPORTED`.

## Operation, privilege, and failure

Policy/trust gates precede native calls and `read-only` blocks mutation. Native
file access is subject to the current token, ACLs, sharing modes, and reparse
point behavior. Child creation uses no shell and is not placed in a Job Object.
Process termination acquires its handle only at enforcement time and remains
subject to target ACL and protected-process rules. Windows Firewall and routing
remain authoritative for TCP. Permission, handle, filesystem, child, socket,
timeout, mismatch, and unavailable cases are reported separately from policy
authorization and completion. Administrators gain OS authority, not stronger
AILEE containment; deploy with least privilege.

## Evidence and limitations

The module was imported and its capability truth table simulated on Linux;
foreign execution correctly reported `UNAVAILABLE`. It was **not natively run
or compiled on Windows in the local acceptance environment**. The repository
configures `windows-latest` to run common and native Local Computing tests, but
no authenticated workflow result was accessible, so Windows CI success is
`NOT VERIFIED` rather than inferred.

Reparse-point and replacement races remain. The adapter creates no restricted
token, Job Object sandbox, AppContainer, WFP filter, service, or driver; it
cannot mediate unrelated operations. Executable/descendant identity is not
transitively bound. A PID is resolved to a handle only during enforcement. DNS
resolution is not policy-pinned, and traffic is not intercepted.

