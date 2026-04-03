# Releasing Ramanujan Tools

This document outlines the standard procedure for publishing a new version of the `ramanujantools` package. 

Our deployment pipeline is split into two automated triggers:
1. **PyPI Publication:** Triggered automatically by pushing a git tag that matches the version in the configuration file.
2. **Zenodo Archival (DOI):** Triggered automatically by creating a formal "Release" on GitHub.

To ensure both platforms are updated correctly, follow these three steps in order.

---

## Step 1: Update the Version File
**What to do:** Create a pull request updating the version string in `pyproject.toml` to the new release number and merge it to `master`.
**Why:** Our Continuous Deployment (CD) pipeline validates that the git tag exactly matches this configuration file. If they do not match, the deployment will fail.

## Step 2: Commit, Tag, and Push
**What to do:** When merged to master, create a git tag for that specific commit (the above merge), and push it to the main repository.

Since we have not published a first official release (i.e, v1.0.0), just bump the patch version (eg. from v0.0.6 to v0.0.7).

```bash
git checkout master
git pull origin master

git tag v0.0.x
git push origin v0.0.x
```

**Why:** Pushing the tag directly via the command line triggers the GitHub Actions CD workflow. The pipeline will run the tag validation check and automatically publish the new package version to PyPI. 

## Step 3: Publish a GitHub Release
**What to do:** Go to the GitHub repository online to formalize the tag into a release.
1. Navigate to the **Releases** page.
2. Click **Draft a new release**.
3. In the **Choose a tag** dropdown, select the `vX.Y.Z` tag you just pushed.
4. Title the release (e.g., "Release vX.Y.Z").
5. Provide a detailed changelog. Do not rely solely on the auto-generated notes; review the merged PRs to ensure all changes are categorized and explained clearly.
6. Click **Publish release**.

**Why:** Zenodo does not monitor raw git tags; it only listens for GitHub webhooks fired by the creation of a formal Release via the UI. Publishing the release tells Zenodo to archive the code tree and mint a new DOI for citation.
