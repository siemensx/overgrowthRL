# AngelScript 2.38.0 source provenance

- Source: `https://www.angelcode.com/angelscript/sdk/files/angelscript_2.38.0.zip`
- Retrieved: 2026-08-15
- SHA-256: `b33b5dbcda10317ef67d628353d83246984ce6fcac102d4dc2aed121eba52e6f`
- Vendored subset: `sdk/angelscript/`

By default, builds continue to use the original AngelScript 2.32.0 tree.
`RL_NATIVE_ARM64_TRAINING` selects this version for the Apple Silicon build;
`RL_ANGELSCRIPT_238` is a default-off experiment switch for equivalence and
performance testing on other targets, including the Windows trainer.

## Local ARM64 correction

`angelscript/source/as_callfunc_arm64.cpp` contains one explicitly marked
Overgrowth patch. AngelScript 2.38.0's `IsRegisterHFAParameter` divided a count
of 32-bit words by a byte size. For a sequence of homogeneous floating-point
aggregates this could admit a ninth floating-point argument register even
though Apple ARM64 provides eight, making `CallARM64` branch to itself forever.
The patch uses `GetSizeInMemoryBytes()` so the register-count units match.
