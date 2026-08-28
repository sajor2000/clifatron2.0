// @ts-check
// CLIFATRON 2.0 — scientific workflow documentation
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'CLIFATRON 2.0',
  tagline: 'One small model → many outcomes → many hospitals → one node',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://sajor2000.github.io',
  baseUrl: '/clifatron2.0/',

  organizationName: 'sajor2000',
  projectName: 'clifatron2.0',

  onBrokenLinks: 'warn',

  // Mermaid diagram support
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/', // docs-only mode: serve docs at site root
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/sajor2000/clifatron2.0/tree/main/website/',
        },
        blog: false, // disable the blog plugin
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      // Render Mermaid nicely in both light and dark mode
      mermaid: {
        theme: {light: 'neutral', dark: 'dark'},
      },
      navbar: {
        title: 'CLIFATRON 2.0',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'workflowSidebar',
            position: 'left',
            label: 'Scientific Workflow',
          },
          {
            href: 'https://github.com/sajor2000/clifatron2.0',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Workflow',
            items: [
              {label: 'Overview', to: '/'},
              {label: 'Data & Tokenization', to: '/data-tokenization'},
              {label: 'Architecture', to: '/architecture'},
              {label: 'Method 3 Wedge', to: '/method3-wedge'},
            ],
          },
          {
            title: 'Project',
            items: [
              {label: 'GitHub', href: 'https://github.com/sajor2000/clifatron2.0'},
              {label: 'CLIFATRON (upstream)', href: 'https://github.com/Common-Longitudinal-ICU-data-Format/CLIFATRON'},
              {label: 'CLIF consortium', href: 'https://github.com/Common-Longitudinal-ICU-data-Format'},
            ],
          },
        ],
        copyright: `CLIFATRON 2.0 — a methods-upgrade layer on CLIFATRON. MIT. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash'],
      },
    }),
};

export default config;
