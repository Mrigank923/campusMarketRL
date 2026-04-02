import "./globals.css";

export const metadata = {
  title: "Campus Market Env",
  description: "Single-container frontend for the campus market RL environment"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
