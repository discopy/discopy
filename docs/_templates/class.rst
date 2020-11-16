{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% set rubric = namespace(printed=false) %}
   {% for item in methods %}
   {% if item not in inherited_members and item != "__init__" %}
   {% if not rubric.printed %}
   .. rubric:: {{ _('Methods') }}

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
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% endif %}

   {% set rubric.printed = true %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endblock %}
