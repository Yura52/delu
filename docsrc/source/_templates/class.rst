{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   .. autosummary::
      :toctree: api

      ~{{ name }}.__init__

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree: api
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: api
   {% for item in Methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% for item in ['__call__', '__enter__', '__exit__', '__len__', '__getitem__', '__getstate__', '__setstate__',] %}
      {% if item in members %}
      ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
