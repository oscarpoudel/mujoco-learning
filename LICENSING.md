# Licensing notes

I checked every license file present in the cloned repositories rather than
assuming. Results as of 2026-11-22, verified against the GitHub API and the
filesystem:

| Path | License | Covers |
|---|---|---|
| `repos/MuJoCo-Tutorial/LICENSE` | **MIT** — Copyright (c) 2022-2026 Manan Tayal | The whole repo |
| `repos/mujoco-robotics-usage` | **No repository license found** | — |
| `repos/mujoco-robotics-usage/asset/panda/LICENSE` | Apache 2.0 | Only the Franka Panda model assets (inherited from MuJoCo Menagerie) |
| `repos/Tutorial_Mujoco_RL` | **No repository license found** | — |
| `repos/Tutorial_Mujoco_RL/1tutorial_ur5e/model/universal_robots_ur5e/LICENSE` | BSD-style (Copyright 2018 ROS Industrial Consortium) | Only the UR5e model |
| `repos/Tutorial_Mujoco_RL/3tutorial_panda_obstacle/model/franka_emika_panda/LICENSE` | Apache 2.0 | Only the Panda model |

The only genuine repository-level license belongs to `MuJoCo-Tutorial`. The
licenses found in the other two cover third-party *robot models*, not the
tutorial code itself — they do not license the authors' Python.

## What "no license" actually means

Under US copyright law and the Berne Convention, a work with no license is
**all rights reserved by the author** by default. GitHub's Terms of Service grant
you the right to view it on GitHub and to fork it on GitHub — nothing more.
There is no implied permission to copy, modify, redistribute, or reuse the code
in your own project.

That is not a theoretical concern here: both unlicensed repos are substantial
original work, and one (`mujoco-robotics-usage`) has clearly invested effort in
its MPC and SLAM implementations.

## What I'm treating as fine

- Cloning into `repos/` for private study, and reading/debugging it locally.
- Running it on my own machine.
- Taking *notes* on the concepts it teaches, and reimplementing those concepts
  myself in `playground/` — ideas and algorithms are not copyrightable, specific
  code is.
- Citing the author when I describe what I learned from them (which the README does).

## What I'm treating as not fine

- Pasting functions or files from `mujoco-robotics-usage` or `Tutorial_Mujoco_RL`
  into `playground/` or any project of mine.
- Redistributing `repos/` — pushing this folder to a public remote, packaging it,
  or sharing the ZIP. Note that `repos/` is therefore in `.gitignore`.
- Assuming the MIT license on `MuJoCo-Tutorial` extends to anything else.

## Before publishing anything built on this

1. Open an issue asking each unlicensed repo's author to add a license. This is
   a routine and usually welcome request; many repos get an MIT or Apache-2.0
   added on request.
2. If they add one, record the version and commit hash here.
3. If they don't, reimplement the idea from scratch and keep the original
   unmodified in `repos/` for reference only.
4. Robot models carry their own terms regardless of tutorial licensing — check
   the model's own `LICENSE` before shipping any derivative work that embeds
   meshes or MJCF from Menagerie.

*I am not a lawyer and this is not legal advice — it's a practical record of
what these repositories do and do not permit.*
