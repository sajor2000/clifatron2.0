// @ts-check
// Ordered sidebar for the CLIFATRON 2.0 scientific workflow.

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  workflowSidebar: [
    'overview',
    'v1-vs-v2',
    'data-tokenization',
    'architecture',
    'objectives-training',
    'method3-wedge',
    'federated-validation',
    'evaluation-panel',
    'ablations',
    {
      type: 'category',
      label: 'Engineering & governance',
      collapsed: false,
      items: [
        'governance-trust',
        'project-status',
      ],
    },
  ],
};

export default sidebars;
