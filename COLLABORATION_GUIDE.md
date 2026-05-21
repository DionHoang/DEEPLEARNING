# Git Collaboration Guide

This guide outlines the recommended Git workflow for contributing to our project, taking into account the following repository rules:

*   **Restrict deletions**: Only authorized personnel with bypass permissions can delete branches (e.g., `main`). This prevents accidental or unauthorized removal of critical project history.
*   **Require a pull request before merging**: Direct pushes to protected branches (e.g., `main`) are disallowed. All changes must be submitted via a Pull Request (PR) to integrate into the main codebase.
*   **Block force pushes**: `git push --force` is strictly prohibited. This rule safeguards the project's commit history, preventing accidental overwrites and ensuring a stable, shared development timeline for all team members.

Following these guidelines will ensure a smooth, secure, and collaborative development process.

## Contributing Guide

### 1. Clone Repository

First, clone the project repository to your local machine:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Pull Latest Code

Before starting any new work, ensure your local `main` branch is up-to-date with the remote repository:

```bash
git checkout main
git pull origin main
```

### 3. Create a New Branch

Create a new feature branch for your work. Use a descriptive name that reflects the purpose of your changes (e.g., `feature/add-user-authentication`, `bugfix/fix-login-issue`, `docs/update-readme`).

```bash
git checkout -b feature/your-feature-name # Example: git checkout -b feature/add-login-page
```

### 4. Commit and Push

Make your changes within your feature branch. Once your work is complete, stage your changes and commit them. It is recommended to commit logical units of work with clear, concise messages.

```bash
# Stage your changes
git add . # Stages all changes in the current directory
# Or to stage specific files/directories:
# git add path/to/your/file.js path/to/your/directory/

# Commit your changes with a descriptive message
git commit -m "feat: Implement new feature X" # Example commit message

# After committing, push your branch to the remote repository
git push origin feature/your-feature-name # Example push command
```

### 5. Create Pull Request

After pushing your branch, navigate to the repository's web interface (e.g., GitHub, GitLab, Bitbucket) and create a new Pull Request. The target branch for your PR should always be `main`.

**Example Pull Request Flow:**

`feature/your-feature-name` → `main`


### 6. Post-Merge Cleanup

Once your Pull Request has been successfully merged into the `main` branch, it is good practice to clean up your local and remote environments.

1.  **Update your local `main` branch:**
    ```bash
    git checkout main
    git pull origin main
    ```
2.  **Delete your local feature branch:**
    ```bash
    git branch -d feature/your-feature-name
    ```
3.  **Delete the remote feature branch:** (This is often done automatically by the Git hosting service after merging, but you can do it manually if needed)
    ```bash
    git push origin --delete feature/your-feature-name
    ```

## Summary of Key Commands

| Command                                   | Description                                                                 |
| :---------------------------------------- | :-------------------------------------------------------------------------- |
| `git clone <url>`                         | Clone a repository into a new directory.                                    |
| `git config user.name "Name"`             | Set the user name for commits.                                              |
| `git config user.email "email"`           | Set the user email for commits.                                             |
| `git checkout main`                       | Switch to the `main` branch.                                                |
| `git pull origin main`                    | Fetch and integrate changes from the remote `main` branch.                  |
| `git checkout -b <branch-name>`           | Create and switch to a new branch.                                          |
| `git add <path>`                          | Stage specific files or directories.                                        |
| `git commit -m "<message>"`             | Record staged changes to the repository with a message.                     |
| `git push origin <branch-name>`           | Push changes to the remote repository.                                      |
| `git branch -d <branch-name>`             | Delete a local branch.                                                      |
| `git push origin --delete <branch-name>`  | Delete a remote branch.                                                     |

By following this workflow, collaborators can effectively contribute to the project while respecting the defined repository rules. If you have any questions, please consult with team lead developers.
