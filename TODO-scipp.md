# TODO

This is the TODO list for the scippification work:

- [ ] `ASEParser`: currently there is no `ASEParser`
- [x] `_get_molecules`: currently the `scipp` kinisi can only calculate diffusion for individual atoms, want to calculate centre of geometry and centre of mass.
- [ ] `save` and `load`: need to be able to save and load `Analyzer` type objects -- needs the `to_dict` and `from_dict` methods for each `Analyzer` type (this should probably be done last as it depends on properties that are present). 
- [ ] `from_*`: currently there is only `from_xdatcar` and `from_universe`, need to add all of the variants in current release version. 
- [x] `posterior_predictive`: add posterior predictive functionality
- [ ] `arrhenius`: the `arrhenius` module is not functional at all (this requires thought about implementation, i.e., do we still want to use `uravu`?).
- [ ] Documentation updates (to happen gradually).
- [ ] tests!
