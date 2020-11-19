{{ '.'.join(fullname.split('.')[1:]) | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :no-inherited-members:
   :member-order: bysource

   {% block methods %}
   {% set rubric = namespace(printed=false) %}
   {% for item in methods %}
   {% if item not in inherited_members and item not in excluded %}
   {% if not rubric.printed %}
   .. rubric:: Methods

   .. autosummary::
   {% endif %}

   {% set rubric.printed = true %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endblock %}

   {% block attributes %}
   {% set rubric = namespace(printed=false) %}
   {% for item in attributes %}
   {% if item not in inherited_members %}
   {% if not rubric.printed %}
   .. rubric:: Attributes

   .. autosummary::
   {% endif %}

   {% set rubric.printed = true %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endblock %}
