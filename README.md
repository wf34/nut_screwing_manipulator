# Nut Screwing Manipulator
This project studies `iiwa` control programs in application to a task of screwing a nut.

## Dependencies
Presently depends on a custom branch of drake, commit hash `d2d229abfeae3073779ab304a0a947bccd14d182`, which is available [here](https://github.com/wf34/drake/commits/experiment/wf34/hybrid_control_for_nut_screwing2). That is due to:
   * `ScrewJoint` functionality not being merged yet to upstream `drake` (e.g. `SdfParser` of `ScrewJoint`)
   * `nut_screwing_manipulator` (unnecessarily) depending on particular commit and branch of `drake` (e.g. hacks, specific to experiment were done in `ManipulationStation`; the `bolt_n_nut.sdf` model was placed within drake)

## Running & Analytics
   * `./build.sh` runs the simulation and stores the telemetry `csv`
       * optional key `--with_external_force` of `run_manipulator` enables `ExternallyAppliedSpatialForce`
       * this script will prompt in the command-line before it proceeds to advance the simulation
   * `./analytics.sh` uses the telemetry csv to build graphs in `png`
