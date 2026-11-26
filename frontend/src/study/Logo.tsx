/** A page with a highlighter stroke across it. */
export default function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" width="30" height="30" aria-hidden="true">
      <rect x="5" y="3" width="22" height="26" rx="4" fill="var(--panel)" stroke="var(--pine)" strokeWidth="2" />
      <rect x="8" y="12" width="16" height="6" rx="1.5" fill="var(--mark)" />
      <path d="M9 8h10M9 22h14" stroke="var(--pine)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
