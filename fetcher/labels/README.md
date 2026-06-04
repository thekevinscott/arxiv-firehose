# labels/

Seed labels for the four classifier prompts. Each subdirectory is a
[coaxer](https://pypi.org/project/coaxer/) label folder: one binary classifier
per category.

## Layout

```
labels/
  <category-slug>/
    _schema.json           # input/output field types + descriptions
    <example-slug>/
      record.json          # {id, inputs, output, meta}
      abstract.txt         # sibling text file referenced by record.json
```

`record.json.inputs.abstract = "abstract.txt"` — coaxer's record loader
substitutes sibling-file contents when the value matches a real file.

## Categories

- `is-about-steering-vector` — activation steering / RepE / SAE-based steering
- `is-image-enhancement` — degraded image → cleaner image (SR, denoise, inpaint, VFI, …)
- `is-survey` — literature surveys
- `is-local-llm-relevant` — techniques that need weight access OR justify local LLMs

Each category's `_schema.json` `output.desc` is the operational definition.
When a paper is borderline, read that field, not the title alone.

## Building compiled prompts

```bash
uvx --from coaxer coax labels/is-about-steering-vector \
  --out prompts/is-about-steering-vector \
  --output-name is_about_steering_vector
```

Repeat for each category. Then point fetcher at the compiled artifacts:

```toml
[classify]
prompts_dirs = [
  "./prompts/is-about-steering-vector",
  "./prompts/is-image-enhancement",
  "./prompts/is-survey",
  "./prompts/is-local-llm-relevant",
]
```

## Seed composition

Per category: ~5 positives, ~5 obvious negatives, ~5–8 ambiguous negatives.
Ambiguous negatives are the hard cases — papers that touch the topic
vocabulary but are not actually about it (e.g., for steering vectors: DPO,
ROME, CFG). The `meta.kind` field records `pos | obvious-neg | ambig-neg`
so you can audit class balance.

## Adding more examples

```bash
mkdir labels/<category>/<short-slug>
$EDITOR labels/<category>/<short-slug>/record.json
$EDITOR labels/<category>/<short-slug>/abstract.txt
```

Then recompile. Coaxer treats each subdirectory as one example; no
manifest to edit.
