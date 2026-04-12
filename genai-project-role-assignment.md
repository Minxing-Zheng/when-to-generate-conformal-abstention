# GenAI Project Role Assignment

| Role | Main Responsibility | Concrete Deliverables | Not Responsible For |
|---|---|---|---|
| Person 1: Environment and Baseline Pipeline Owner | Set up the reusable environment and run the first end-to-end baseline on a non-trivial dataset scale | `Conda` environment or `Apptainer` setup, minimal runnable generation/scoring pipeline, standardized input/output format, and a normal-scale `Top-1` baseline result that can go into the midway report | Pre-configuring every future experiment, running all later ablations, or owning all threshold/conformal experiments |
| Person 2: Selective Generation and Evaluation Owner | Build the selection logic and run the method-side experiments on top of the existing pipeline | `naive threshold` baseline, conformal calibration/abstention logic, `SelQual` and `AccRate` evaluation, and comparison tables/figures | Setting up the full model environment from scratch or owning low-level infrastructure/debugging |
| Person 3: Experiment Analysis and Report Owner | Organize experiment outputs into a coherent midway submission and own result aggregation/visualization | Midway report draft, skeleton tables/plots, result aggregation, plotting scripts, qualitative examples/failure-case analysis, timeline/plan section, and integrated write-up of results from Persons 1 and 2 | Owning the full modeling pipeline or independently implementing the core selective-generation method |

Persons 1 and 2 will each keep records of runtime, hardware, and resource usage for their own experiments, and Person 3 will compile these records into the report.
