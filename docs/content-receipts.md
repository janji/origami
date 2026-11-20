# Explicit content verification

`origami content` verifies an **explicit ordered list of files** against expected SHA256 digests. It does not load a model, use the GPU, start a service, or change prompt-cache identity. Reading a large checkpoint can take a long time and consume storage bandwidth; perform it in an approved maintenance window.

The expected hashes must come from a source you trust and pin independently. The tool proves equality with the supplied manifest. It does **not** authenticate a publisher, certify model safety, or discover whether the manifest includes every weight, configuration, tokenizer, template, external GGUF or expert file used by a model. Include all relevant files before describing the identity as covering that model. A declared hash is never treated as evidence without reading the bytes.

A manifest has this schema (replace the example digest with the actual expected digest):

```json
{
  "schemaVersion": 1,
  "algorithm": "sha256",
  "files": [
    {
      "path": "model-00001-of-00002.safetensors",
      "size": 123456,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
```

Paths must be unique relative names with no empty, `.` or `..` components. Digests use 64 lowercase hexadecimal characters. The file order is significant; preserve the same order when transporting a manifest. An empty file is allowed with its actual empty-file digest. At most 4,096 files and 8 MiB of manifest/receipt metadata are accepted.

```sh
origami content verify pinned-manifest.json --root /path/to/model --receipt /path/to/receipts/model.json
origami content check /path/to/receipts/model.json --json
origami content import /path/to/receipts/model.json --root /path/to/copied-model --receipt /path/to/receipts/copied-model.json
```

The receipt parent directory must already exist. Verification streams payloads in chunks of at most 8 MiB, compares every size and digest, checks open-file and path incarnations before and after reading, and rechecks all earlier files before publication. A receipt binds the portable ordered content identity to the local resolved paths, device/inode, size, and nanosecond modification/change/birth times. Rewriting bytes while restoring mtime or replacing a same-sized file invalidates that binding. Symlink targets are resolved and bound; retargeting them invalidates the receipt. Symlinks may point outside the root, so inspect manifest paths and their targets before authorizing a full read.

`check` reads receipt metadata and file stats only. It returns `incarnation_bindings_match`, not a new content-verification claim. Receipts are trusted local records, **not signed attestations**: protect them from untrusted edits. A filesystem that forges or reuses all bound metadata is outside this mechanism's guarantee. Files can still change after a check; this tool does not freeze them or grant a serving process ownership. No automatic cache reuse is enabled by these commands.

`import` always rehashes every file against the receipt's embedded expected digests. Copying a receipt or trusting its declared hashes is insufficient. Identical ordered files retain the portable identity after successful re-verification, but receive new local bindings. Changing shard order changes identity even if the set of files is the same.

Publication writes an owner-only staging file beside the destination, synchronizes it, and atomically renames it. A mismatch, cancellation, mutation, or write failure before publication preserves an existing valid receipt. Handled failures remove staging; a process killed without cleanup can leave a `.origami-content-*.tmp` file, which is never accepted as the destination receipt. Rename atomicity prevents partial JSON; best-effort directory sync is not a guarantee against sudden storage power loss. The output cannot replace a listed payload (including a hard link to one), a symlink, or an unrelated existing file. Existing receipt files can be replaced.
