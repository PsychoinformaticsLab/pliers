Developer guide
===============

Pliers is, at its core, a standardized interface to other feature extraction tools and services. This means that, to a greater extent than most other open-source projects, the utility of pliers is closely tied to the number (and quality) of community-contributed extensions. 

The easiest way to contribute new functionality to pliers is to create new |Stim| or |Transformer| classes.

Creating new |Stim| classes
---------------------------

Every |Stim| in pliers must inherit from the base |Stim| class or one of its subclasses. 