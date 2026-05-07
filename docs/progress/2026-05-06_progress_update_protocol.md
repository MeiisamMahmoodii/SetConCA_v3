# Task: Progress Update Protocol

Tags: #progress #documentation #workflow #session

Related notes: [[README]] [[2026-05-06_progress_system_setup]] [[2026-05-06_raw_json_to_dataset_guide]]

## 1. Goal

The user clarified that from this point forward, they can simply say "update" and the project progress documentation should be updated based on the current conversation and work.

The user also provided this conversation/session reference:

```text
019dfb85-1c81-74f1-9bfe-ffa587d400ad
```

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-06 |
| Workspace | `C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2` |
| Progress root | `docs/progress` |
| Session reference | `019dfb85-1c81-74f1-9bfe-ffa587d400ad` |

## 3. Hypothesis Or Rationale

The progress log should be easy to maintain during active research. Requiring the user to repeat the full documentation instruction every time would slow the project down. A short "update" command should be enough to trigger documentation updates using the current conversation, code changes, outputs, figures, failures, and reasoning.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Recorded the user's update instruction. | Needed to preserve the workflow rule in the project notebook. | This note now defines the update protocol. | Succeeded |
| 2 | Recorded the session reference. | Needed to connect future updates to the provided conversation identifier. | Session reference saved above. | Succeeded |
| 3 | Updated the main progress index. | Needed to make the protocol discoverable from [[README]]. | Added this note to the task index and workflow section. | Succeeded |

## 5. Operating Rule From Now On

When the user says `update`, update the relevant progress documentation using the current conversation and project state.

Depending on what changed, this may include:

- Creating a new task note.
- Updating an existing task note.
- Updating a guide.
- Adding commands, code snippets, pseudocode, tables, figures, results, and failure analysis.
- Adding paper/tool references when outside work or new technology is used.
- Updating [[README]] with new Obsidian links.

## 6. Code And Pseudocode

```text
if user_message means "update progress docs":
    inspect recent conversation and local changes
    identify the relevant task or guide
    if task has no note:
        create docs/progress/YYYY-MM-DD_short_task.md
    else:
        append or revise the existing note
    include:
        goal
        actions
        reasoning
        results
        failures/limits
        files changed
        external papers/tools
        follow-up
    update docs/progress/README.md links
```

## 7. Results

| Result | Value |
| --- | --- |
| Update protocol recorded | Yes |
| Session reference recorded | Yes |
| Main progress index updated | Yes |

## 8. Interpretation

This converts the progress system from a static folder into an active lab notebook workflow. The user can keep working naturally, then ask for an update when they want the documentation brought in sync.

## 9. Failures Or Limits

The assistant can only update based on visible conversation context and files available in the workspace. If work happened outside the visible workspace or outside the current conversation, the user should point to the files, outputs, or result folders that should be included.

## 10. External Works And Papers

No external papers, algorithms, datasets, or new technologies were used for this protocol update.

## 11. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `docs/progress/2026-05-06_progress_update_protocol.md` | Added protocol note. | Preserve the user's update instruction and session reference. |
| `docs/progress/README.md` | Added index entry and workflow rule. | Make the protocol discoverable. |

## 12. Follow-Up

- [ ] Use this protocol whenever the user says `update`.
- [ ] Keep new progress notes connected with Obsidian links.

