{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :show-inheritance:

.. include:: backreferences/{{ fullname }}.examples

.. raw:: html

     <div style='clear:both'></div>

Attributes
^^^^^^^^^^

{% for item in attributes %}
.. autoattribute:: {{ objname }}.{{ item }}
{% endfor %}


Methods
^^^^^^^

{% for item in methods %}
{% if item != '__init__' %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}
{% endfor %}

.. raw:: html

     <div style='clear:both'></div>
