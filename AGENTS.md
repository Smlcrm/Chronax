## README and citation

Invoke `simulacrum-readme` on repository creation and on significant change.

- Repository creation: before the first commit that is pushed.
- Significant change: a public API or CLI command is added, removed or renamed;
  install steps or dependencies change; a major or minor version bump; a
  license change; a repository rename; a new release tag.
- Not for typo fixes, internal refactors or test-only changes.

The `readme-standard` workflow (`.github/workflows/smlcrm-readme.yml`) fails a
pull request whose README.md or CITATION.cff is missing a required section.
Text between `<!-- smlcrm:begin NAME -->` and `<!-- smlcrm:end NAME -->` is
owned by the skill; edit outside the markers.
