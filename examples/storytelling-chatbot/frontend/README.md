This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.

## Dependency compatibility

Use Node.js 20.19 or later and `npm ci` to reproduce the checked-in dependency tree.
Run `npm run lint`, `npm run build`, and `npm audit` before updating dependencies.

The app uses Next.js 15 and Tailwind CSS 4. Theme, animation-plugin and container
configuration now live in `app/globals.css`; CSS modules that use `@apply` reference
that stylesheet. Tailwind 4 requires Safari 16.4+, Chrome 111+ or Firefox 128+.
Daily React uses its Jotai-based implementation because the older Recoil-based
provider cannot render with the React runtime bundled by the Next.js App Router.
The Daily room creation, device selection, join and leave flow is preserved.

The ESLint rule configuration intentionally remains at `eslint-config-next@14.2.35`.
Its Next/React/TypeScript rules work with the Next.js 15 lint command; the newer
configuration introduces `fast-glob` and its unpatched `braces` stack-exhaustion
advisory. The `glob` override stays within major version 10 to patch that rule
configuration's pinned dependency, and the PostCSS override uses the same patched
PostCSS 8 release for Next.js and Tailwind. Recheck upstream dependencies before
removing either override or upgrading the rule configuration.

This example exports static files. As before, Next.js rewrites apply to the
development server; a static production host must proxy room-creation requests to
the storytelling server separately.
